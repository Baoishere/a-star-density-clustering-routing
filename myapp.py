import streamlit as st
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import heapq
from sklearn.neighbors import KDTree

# =============================================================================
# CÁC HÀM TỪ NOTEBOOK
# =============================================================================

@st.cache_data
def load_data():
    """Load dữ liệu từ file CSV"""
    try:
        df = pd.read_csv("processed_data_with_hdbscan.csv", parse_dates=["date"])
        return df
    except Exception as e:
        st.error(f"Lỗi khi load dữ liệu: {e}")
        return pd.DataFrame()

def get_cluster_congestion_cost(df, cluster_column='hdbscan_cluster'):
    """Tính chi phí ùn tắc cho mỗi cụm dựa trên LOS trung bình"""
    los_weights = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 10}
    
    # Tính điểm LOS trung bình cho mỗi cụm
    cluster_stats = df[df[cluster_column] != -1].groupby(cluster_column).agg({
        'LOS': lambda x: np.mean([los_weights.get(los, 3) for los in x]),
        'center_lat': 'mean',
        'center_lng': 'mean'
    }).rename(columns={'LOS': 'avg_congestion_score'})
    
    # Chuẩn hóa điểm ùn tắc thành chi phí (1-10)
    min_score = cluster_stats['avg_congestion_score'].min()
    max_score = cluster_stats['avg_congestion_score'].max()
    
    if max_score > min_score:
        cluster_stats['travel_cost'] = 1 + (cluster_stats['avg_congestion_score'] - min_score) / (max_score - min_score) * 9
    else:
        cluster_stats['travel_cost'] = 1

    # Add congestion_ratio for popup
    cluster_stats['congestion_ratio'] = cluster_stats['avg_congestion_score'] / max_score if max_score > 0 else 0

    return cluster_stats[['travel_cost', 'center_lat', 'center_lng', 'congestion_ratio']].to_dict('index')

def calculate_cluster_centers(df, cluster_column='hdbscan_cluster'):
    """Tính toán tâm của các cụm"""
    cluster_centers = df.groupby(cluster_column).agg({
        'center_lat': 'mean',
        'center_lng': 'mean'
    }).reset_index()

    # Loại bỏ cụm nhiễu (-1)
    cluster_centers = cluster_centers[cluster_centers[cluster_column] != -1]

    return cluster_centers

def calculate_cluster_adjacency(df, cluster_column='hdbscan_cluster', distance_threshold=0.05):
    """Tính toán ma trận kề giữa các cụm dựa trên khoảng cách"""
    cluster_centers = calculate_cluster_centers(df, cluster_column)
    cluster_adjacency = {}

    for i, cluster1 in cluster_centers.iterrows():
        cluster_id1 = cluster1[cluster_column]
        lat1, lng1 = cluster1['center_lat'], cluster1['center_lng']
        cluster_adjacency[cluster_id1] = []

        for j, cluster2 in cluster_centers.iterrows():
            if i != j:
                cluster_id2 = cluster2[cluster_column]
                lat2, lng2 = cluster2['center_lat'], cluster2['center_lng']

                # Tính khoảng cách Euclidean
                distance = np.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2)

                if distance <= distance_threshold:
                    cluster_adjacency[cluster_id1].append(cluster_id2)

    return cluster_adjacency

def heuristic_cluster(cluster1, cluster2, cluster_info):
    """Hàm heuristic cho A* - khoảng cách Euclidean giữa các cụm"""
    if cluster1 not in cluster_info or cluster2 not in cluster_info:
        return float('inf')

    lat1, lng1 = cluster_info[cluster1]['center_lat'], cluster_info[cluster1]['center_lng']
    lat2, lng2 = cluster_info[cluster2]['center_lat'], cluster_info[cluster2]['center_lng']

    return np.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2)

def find_cluster_for_point(lat, lng, df, cluster_column='hdbscan_cluster'):
    """Tìm cụm gần nhất cho một điểm tọa độ"""
    # Loại bỏ cụm nhiễu
    valid_clusters = df[df[cluster_column] != -1]

    if valid_clusters.empty:
        return None

    # Tính khoảng cách đến tâm của mỗi cụm
    distances = []
    for cluster_id in valid_clusters[cluster_column].unique():
        cluster_data = valid_clusters[valid_clusters[cluster_column] == cluster_id]
        center_lat = cluster_data['center_lat'].mean()
        center_lng = cluster_data['center_lng'].mean()

        distance = np.sqrt((lat - center_lat)**2 + (lng - center_lng)**2)
        distances.append((cluster_id, distance))

    # Trả về cụm gần nhất
    if distances:
        return min(distances, key=lambda x: x[1])[0]
    return None

def a_star_cluster_based(start_lat, start_lng, goal_lat, goal_lng, df, cluster_column='hdbscan_cluster'):
    """Thuật toán A* sử dụng các cụm làm nút"""

    # Tìm cụm cho điểm bắt đầu và kết thúc
    start_cluster = find_cluster_for_point(start_lat, start_lng, df, cluster_column)
    goal_cluster = find_cluster_for_point(goal_lat, goal_lng, df, cluster_column)

    if start_cluster is None or goal_cluster is None:
        st.error("Không thể tìm thấy cụm cho điểm bắt đầu hoặc kết thúc")
        return []

    # Tính toán thông tin cụm và ma trận kề
    cluster_info = get_cluster_congestion_cost(df, cluster_column)
    cluster_adjacency = calculate_cluster_adjacency(df, cluster_column)

    # Khởi tạo A*
    open_set = []
    heapq.heappush(open_set, (0, start_cluster))

    came_from = {}
    g_score = {cluster: float('inf') for cluster in cluster_info.keys()}
    g_score[start_cluster] = 0

    f_score = {cluster: float('inf') for cluster in cluster_info.keys()}
    f_score[start_cluster] = heuristic_cluster(start_cluster, goal_cluster, cluster_info)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_cluster:
            # Tái tạo đường đi
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor in cluster_adjacency.get(current, []):
            if neighbor not in cluster_info:
                continue

            # Chi phí = khoảng cách + chi phí ùn tắc
            base_cost = heuristic_cluster(current, neighbor, cluster_info)
            congestion_cost = cluster_info[neighbor]['travel_cost']
            total_cost = base_cost * congestion_cost

            tentative_g = g_score[current] + total_cost

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic_cluster(neighbor, goal_cluster, cluster_info)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

def find_streets_by_name(df, street_name):
    """Tìm các đường có tên chứa chuỗi nhập vào"""
    street_name_lower = street_name.lower()
    matching_streets = df[df['street_name'].str.lower().str.contains(street_name_lower, na=False)]

    if matching_streets.empty:
        return None

    # Lấy danh sách các đường duy nhất
    unique_streets = matching_streets['street_name'].unique()
    return unique_streets

def get_street_coordinates(df, street_name):
    """Lấy tọa độ trung bình của một đường"""
    street_data = df[df['street_name'] == street_name]

    if street_data.empty:
        return None, None

    # Tính tọa độ trung bình của đường
    avg_lat = street_data['center_lat'].mean()
    avg_lng = street_data['center_lng'].mean()

    return avg_lat, avg_lng

def create_cluster_path_visualization(cluster_path, df, start_lat, start_lng, goal_lat, goal_lng, cluster_column='hdbscan_cluster'):
    """Tạo visualization cho đường đi qua các cụm - CHỈ hiển thị các cụm được đi qua"""

    # Lấy thông tin về các cụm
    cluster_info = get_cluster_congestion_cost(df, cluster_column)

            # Tạo bản đồ
    center_lat = (start_lat + goal_lat) / 2
    center_lng = (start_lng + goal_lng) / 2
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13)

    # CHỈ vẽ các cụm có trong đường đi (cluster_path)
    path_coords = []
    for cluster_id in cluster_path:
        if cluster_id in cluster_info:
            info = cluster_info[cluster_id]
            color = 'green' if info['travel_cost'] <= 3 else 'orange' if info['travel_cost'] <= 6 else 'red'

            # Vẽ cụm
            folium.CircleMarker(
                location=[info['center_lat'], info['center_lng']],
                radius=10 + info['travel_cost'] * 2,
                popup=f'Cụm {cluster_id}<br>Chi phí: {info["travel_cost"]:.2f}<br>Ùn tắc: {info["congestion_ratio"]:.1%}',
                color=color,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(m)

            # Thêm tọa độ vào đường đi
            lat, lng = info['center_lat'], info['center_lng']
            path_coords.append([lat, lng])

    # Vẽ đường đi qua các cụm
    if path_coords:
        # Vẽ đường nối các cụm
        folium.PolyLine(
            path_coords,
            color='blue',
            weight=4,
            opacity=0.8,
            popup='Đường đi tối ưu qua các cụm'
        ).add_to(m)

        # Thêm mũi tên chỉ hướng
        for i in range(len(path_coords) - 1):
            start_point = path_coords[i]
            end_point = path_coords[i + 1]

            # Tính điểm giữa để đặt mũi tên
            mid_point = [
                (start_point[0] + end_point[0]) / 2,
                (start_point[1] + end_point[1]) / 2
            ]

            # Tạo mũi tên
            arrow = folium.RegularPolygonMarker(
                location=mid_point,
                number_of_sides=3,
                radius=8,
                rotation=np.degrees(np.arctan2(end_point[1]-start_point[1], end_point[0]-start_point[0])),
                color='blue',
                fillColor='blue',
                fillOpacity=0.8
            )
            arrow.add_to(m)

    # Đánh dấu điểm bắt đầu và kết thúc
    folium.Marker(
        [start_lat, start_lng],
        popup='Điểm bắt đầu'
    ).add_to(m)

    folium.Marker(
        [goal_lat, goal_lng],
        popup='Điểm kết thúc'
    ).add_to(m)

    return m

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    # Cấu hình trang
    st.set_page_config(
        page_title="Hệ Thống Tìm Đường Tránh Ùn Tắc",
        layout="wide"
    )
    
    # Tiêu đề app
    st.title("Hệ Thống Tìm Đường Tránh Ùn Tắc")
    
    # Load dữ liệu
    with st.spinner("Đang tải dữ liệu..."):
        df = load_data()
        
        if df.empty:
            st.error("Không thể tải dữ liệu. Vui lòng kiểm tra đường dẫn file.")
            return
    
    # Sidebar - Chọn điểm đi và đến bằng tên đường
    with st.sidebar:
        st.header("Chọn Điểm Đi và Đến")
        
        # Chọn đường cho điểm đi (1 ô duy nhất, có thể gõ để nhảy đến lựa chọn)
        
        all_streets = sorted(df['street_name'].dropna().unique())
        start_street = st.selectbox("Chọn đường đi:", all_streets, key="start_street_select")
        
        # Chọn đường cho điểm đến (1 ô duy nhất, có thể gõ để nhảy đến lựa chọn)
        st.subheader("Điểm đến")
        end_street = st.selectbox("Chọn đường đến:", all_streets, key="end_street_select")
        
        # Nút tìm đường
        find_route = st.button("Tìm Đường Đi", type="primary", use_container_width=True)
    
    # Main content - Hiển thị bản đồ
    col1 = st.columns([2])[0]
    
    with col1:
        
        # Show route when button pressed, or restore last route from session_state
        session_has_path = 'cluster_path' in st.session_state and st.session_state.get('cluster_path')

        if find_route:
            if not start_street or not end_street:
                st.error("Vui lòng chọn cả điểm đi và điểm đến")
            else:
                with st.spinner("Đang tìm đường đi tối ưu..."):
                    try:
                        # Lấy tọa độ từ tên đường
                        start_lat, start_lng = get_street_coordinates(df, start_street)
                        end_lat, end_lng = get_street_coordinates(df, end_street)

                        if start_lat is None or start_lng is None:
                            st.error(f"Không thể xác định tọa độ cho đường: {start_street}")
                        elif end_lat is None or end_lng is None:
                            st.error(f"Không thể xác định tọa độ cho đường: {end_street}")
                        else:

                            # Xác định cụm cho điểm bắt đầu và kết thúc (debug)
                            start_cluster_debug = find_cluster_for_point(start_lat, start_lng, df)
                            end_cluster_debug = find_cluster_for_point(end_lat, end_lng, df)

                            # Tìm đường đi (A* dựa trên cụm)
                            cluster_path = a_star_cluster_based(start_lat, start_lng, end_lat, end_lng, df)

                            if cluster_path:
                                # Lưu vào session_state để giữ map qua các rerun
                                st.session_state['cluster_path'] = cluster_path
                                st.session_state['start_lat'] = start_lat
                                st.session_state['start_lng'] = start_lng
                                st.session_state['end_lat'] = end_lat
                                st.session_state['end_lng'] = end_lng
                                # Use different keys than widget keys to avoid Streamlit error
                                st.session_state['start_street_selected'] = start_street
                                st.session_state['end_street_selected'] = end_street
                                st.session_state['start_cluster_debug'] = start_cluster_debug
                                st.session_state['end_cluster_debug'] = end_cluster_debug

                                # Tạo bản đồ
                                map_viz = create_cluster_path_visualization(
                                    cluster_path, df, start_lat, start_lng, end_lat, end_lng
                                )

                                # Hiển thị bản đồ (full width)
                                components.html(map_viz._repr_html_(), height=700, scrolling=True)

                                # Hiển thị thông tin đường đi
                                st.success(f"Đã tìm thấy đường đi qua {len(cluster_path)} cụm")

                                # Tính toán thống kê
                                cluster_info = get_cluster_congestion_cost(df)
                                total_cost = sum(cluster_info[cluster_id]['travel_cost'] for cluster_id in cluster_path if cluster_id in cluster_info)
                                avg_congestion = np.mean([cluster_info[cluster_id]['congestion_ratio'] for cluster_id in cluster_path if cluster_id in cluster_info])

                                with st.expander("Thông tin chi tiết đường đi"):
                                    st.write(f"**Đường đi qua các cụm:** {' → '.join(map(str, cluster_path))}")
                                    st.write(f"**Tổng chi phí:** {total_cost:.2f}")
                                    st.write(f"**Tỷ lệ ùn tắc trung bình:** {avg_congestion:.1%}")

                            else:
                                # Clear stored path if search failed
                                if 'cluster_path' in st.session_state:
                                    del st.session_state['cluster_path']
                                st.error("Không tìm thấy đường đi phù hợp")

                    except Exception as e:
                        st.error(f"Lỗi khi tìm đường: {e}")

        elif session_has_path:
            # Restore and show last found path from session_state so map doesn't reset
            cluster_path = st.session_state['cluster_path']
            start_lat = st.session_state['start_lat']
            start_lng = st.session_state['start_lng']
            end_lat = st.session_state['end_lat']
            end_lng = st.session_state['end_lng']

            map_viz = create_cluster_path_visualization(
                cluster_path, df, start_lat, start_lng, end_lat, end_lng
            )
            components.html(map_viz._repr_html_(), height=700, scrolling=True)

            st.success(f"Đã hiển thị đường đi đã tìm trước đó (qua {len(cluster_path)} cụm)")

            # Tính toán thống kê (dùng dữ liệu đã lưu)
            cluster_info = get_cluster_congestion_cost(df)
            total_cost = sum(cluster_info[cluster_id]['travel_cost'] for cluster_id in cluster_path if cluster_id in cluster_info)
            avg_congestion = np.mean([cluster_info[cluster_id]['congestion_ratio'] for cluster_id in cluster_path if cluster_id in cluster_info])

            with st.expander("Thông tin chi tiết đường đi"):
                st.write(f"**Đường đi qua các cụm:** {' → '.join(map(str, cluster_path))}")
                st.write(f"**Tổng chi phí:** {total_cost:.2f}")
                st.write(f"**Tỷ lệ ùn tắc trung bình:** {avg_congestion:.1%}")

        else:
            # Hiển thị bản đồ mặc định khi chưa tìm đường
            m = folium.Map(location=[10.7769, 106.7009], zoom_start=12)
            components.html(m._repr_html_(), height=600, scrolling=True)
            st.info("Chọn điểm đi và đến ở sidebar rồi nhấn 'Tìm Đường Đi'")

if __name__ == "__main__":
    main()