import torch
import numpy as np

def dist_to_line_original(x, normal, point):
    """
    Original implementation from the notebook (fixed for broadcasting)
    x: [N, 2]
    normal: [K, 2]
    point: [K, 2]
    return: [N, K]
    """
    return (x.unsqueeze(1) - point).matmul(normal.t().unsqueeze(0)).squeeze(-1) / torch.norm(normal, dim=1).unsqueeze(0)

def dist_to_line_corrected(x, normal, point):
    """
    Corrected implementation
    x: [N, 2]
    normal: [K, 2] 
    point: [K, 2]
    return: [N, K]
    """
    # Compute dot product: (x - point) · normal for each x and each line
    dot_products = torch.sum((x.unsqueeze(1) - point.unsqueeze(0)) * normal.unsqueeze(0), dim=-1)
    # Take absolute value and normalize by normal magnitude
    return torch.abs(dot_products) / torch.norm(normal, dim=1).unsqueeze(0)

def test_dist_to_line():
    """Test the distance to line function with known geometric cases"""
    
    # Test case 1: Point on the line should have distance 0
    print("Test 1: Point on line")
    x = torch.tensor([[0.0, 0.0]])  # Point at origin
    normal = torch.tensor([[1.0, 0.0]])  # Horizontal line normal (vertical line)
    point = torch.tensor([[0.0, 0.0]])  # Line passes through origin
    
    dist_orig = dist_to_line_original(x, normal, point)
    dist_corr = dist_to_line_corrected(x, normal, point)
    
    print(f"Original: {dist_orig}")
    print(f"Corrected: {dist_corr}")
    print(f"Expected: 0.0")
    print()
    
    # Test case 2: Point at distance 1 from line
    print("Test 2: Point at distance 1 from vertical line")
    x = torch.tensor([[1.0, 0.0]])  # Point at (1, 0)
    normal = torch.tensor([[1.0, 0.0]])  # Normal pointing right (vertical line)
    point = torch.tensor([[0.0, 0.0]])  # Line is x = 0 (y-axis)
    
    dist_orig = dist_to_line_original(x, normal, point)
    dist_corr = dist_to_line_corrected(x, normal, point)
    
    print(f"Original: {dist_orig}")
    print(f"Corrected: {dist_corr}")
    print(f"Expected: 1.0")
    print()
    
    # Test case 3: Multiple points and lines
    print("Test 3: Multiple points and lines")
    x = torch.tensor([[0.0, 1.0], [2.0, 0.0], [1.0, 1.0]])  # 3 points
    normal = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # 2 normals (horizontal and vertical lines)
    point = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # Both lines pass through origin
    
    dist_orig = dist_to_line_original(x, normal, point)
    dist_corr = dist_to_line_corrected(x, normal, point)
    
    print(f"Original shape: {dist_orig.shape}")
    print(f"Original:\n{dist_orig}")
    print(f"Corrected shape: {dist_corr.shape}")
    print(f"Corrected:\n{dist_corr}")
    
    # Expected distances:
    # Point (0,1) to horizontal line y=0: distance = 1
    # Point (0,1) to vertical line x=0: distance = 0
    # Point (2,0) to horizontal line y=0: distance = 0  
    # Point (2,0) to vertical line x=0: distance = 2
    # Point (1,1) to horizontal line y=0: distance = 1
    # Point (1,1) to vertical line x=0: distance = 1
    expected = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    print(f"Expected:\n{expected}")
    print()
    
    # Test case 4: Negative distances (should be absolute)
    print("Test 4: Points on both sides of line")
    x = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])  # Points on both sides of y-axis
    normal = torch.tensor([[1.0, 0.0]])  # Normal pointing right (vertical line x=0)
    point = torch.tensor([[0.0, 0.0]])  # Line passes through origin
    
    dist_orig = dist_to_line_original(x, normal, point)
    dist_corr = dist_to_line_corrected(x, normal, point)
    
    print(f"Original: {dist_orig}")
    print(f"Corrected: {dist_corr}")
    print(f"Expected: [[1.0], [1.0]] (both should be positive)")
    print()

def test_with_non_unit_normals():
    """Test with non-unit normal vectors"""
    print("Test 5: Non-unit normal vectors")
    x = torch.tensor([[2.0, 0.0]])  # Point at (2, 0)
    normal = torch.tensor([[2.0, 0.0]])  # Non-unit normal (length 2)
    point = torch.tensor([[0.0, 0.0]])  # Line passes through origin
    
    dist_orig = dist_to_line_original(x, normal, point)
    dist_corr = dist_to_line_corrected(x, normal, point)
    
    print(f"Original: {dist_orig}")
    print(f"Corrected: {dist_corr}")
    print(f"Expected: 2.0 (distance should be independent of normal magnitude)")
    print()

if __name__ == "__main__":
    print("Testing dist_to_line function implementations")
    print("=" * 50)
    
    test_dist_to_line()
    test_with_non_unit_normals()
    
    print("Test complete!")