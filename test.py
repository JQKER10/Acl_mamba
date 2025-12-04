import torch
from model.mamba2_model import MAMBA2   # đổi thành đúng tên file/module

def smoke_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # cấu hình giống lúc khởi tạo model thật
    img_size = (256, 256)
    in_channels = 1
    out_channels = 2
    neighbor_slices = 2
    B = 1

    model = MAMBA2(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        dims=(64, 128, 256, 512),
        depths=(2, 2, 2, 2),
        num_heads=(2, 4, 8, 8),
        neighbor_slices=neighbor_slices,
        deep_supervision=True,   # hoặc False
    ).to(device)

    H, W = img_size

    # input giả
    x_center = torch.randn(B, in_channels, H, W, device=device)
    x_neighbors = torch.randn(B, neighbor_slices, in_channels, H, W, device=device)

    model.eval()
    with torch.no_grad():
        out = model(x_center, x_neighbors)

    # nếu deep_supervision=True, out có dạng (main, [ds...])
    if isinstance(out, tuple):
        main_out, ds_outs = out
        print("Main output shape:", main_out.shape)
        for i, y in enumerate(ds_outs):
            print(f"DS[{i}] shape:", y.shape)
    else:
        print("Output shape:", out.shape)

if __name__ == "__main__":
    smoke_test()
