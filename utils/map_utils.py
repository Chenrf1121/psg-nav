from skimage.measure import find_contours

def get_frontiers(traversible):
    """
    提取所有边界，并用每条边界的首尾两个点表示。
    
    参数:
        traversible (ndarray): 可通行性地图，0 表示可通行，1 表示障碍物。
    
    返回:
        boundaries (list): 每条边界的首尾点列表，格式为 [(start1, end1), (start2, end2), ...]。
    """
    # 找到所有边界
    contours = find_contours(traversible, level=0.5)
    
    boundaries = []
    for contour in contours:
        # 每条边界的首尾点
        start = tuple(contour[0].astype(int))  # 首点
        end = tuple(contour[-1].astype(int))  # 尾点
        boundaries.append((start, end))
    
    return boundaries