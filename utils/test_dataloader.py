import time

from dataset_conversion.BraTsData_person import get_dataloader
from tqdm import  tqdm
from swap_dimensions import swap_batch_slice_dimensions
from show_image import show_mask_origin

if __name__ == '__main__':
    root_dir = "E:\Work\dataset\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    slice_deep = 128
    batch_size = 16
    index = 0
    dataloader = get_dataloader(root_dir=root_dir, batch_size=16, slice_deep=128, slice_size=192,
                                mask_kernel_size=3, binary_mask='1000', mask_rate=0.75,
                                num_workers=4, mode='test')
    loop = 2
    total = 1251 * batch_size
    print("Total: ", total)
    current = 0

    current_p = 0
    start = time.time()
    with tqdm(dataloader) as pbar:
        for (x, y), path in pbar:
            for step in range(slice_deep // batch_size):
                print("Step: ", step)
                print(x.shape, y.shape)
            # x = swap_batch_slice_dimensions(x)
            # y = swap_batch_slice_dimensions(y)
            # print(x.shape, y.shape)
            # show_mask_origin(x, y, index)
            current_p += x.shape[0]
            current += 32
            # if i == loop:
            #     break
    end = time.time()
    print("Time: ", end - start)

    print("Current: ", current)
    print("Current_p: ", current_p)
