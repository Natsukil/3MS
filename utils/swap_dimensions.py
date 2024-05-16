def swap_batch_slice_dimensions(tensor):
    """
    交换一个4D张量中第一维度（批次大小）和第二维度（切片数）。输入张量应具有形状 [batch_size, slice_num, height, width]。
    args:
    tensor (torch.Tensor): 形状为 [batch_size, slice_num, height, width] 的4D张量。
    return:
    torch.Tensor: 形状为 [slice_num, batch_size, height, width] 的4D张量
    """
    # Using permute to swap the dimensions
    swapped_tensor = tensor.permute(1, 0, 2, 3)
    return swapped_tensor