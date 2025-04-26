# Báo cáo: Quantum Neural Network (QNN) cho CIFAR-10

## Mô tả giải pháp (QNN ban đầu)

Giải pháp ban đầu triển khai một **Quantum Neural Network (QNN)** lai (hybrid quantum-classical) để phân loại hình ảnh trên tập dữ liệu CIFAR-10 (10 lớp, hình ảnh 32x32x3). Các thành phần chính của giải pháp bao gồm:

-   **Kiến trúc mạng**:
    -   **Tầng tích chập cổ điển**: Một tầng `Conv2d` (3->16 kênh, kernel 3x3, stride 2) để trích xuất đặc trưng không gian, giảm kích thước ảnh từ 32x32 xuống 15x15.
    -   **Tầng tiền xử lý (pre_net)**: Tầng fully connected (Linear) ánh xạ từ 16*15*15 (3600 chiều) xuống 4 chiều (n_qubits=4), cung cấp đầu vào cho mạch lượng tử.
    -   **Mạch lượng tử**: Mạch lượng tử sử dụng 4 qubit, 2 tầng (n_layers=2), với:
        -   Mã hóa dữ liệu bằng cổng `RX` cho từng qubit (dữ liệu đầu vào được ánh xạ trực tiếp).
        -   Cấu trúc mạch gồm cổng `RY` (tham số học được) và `CZ` để tạo rối lượng tử.
        -   Đo giá trị kỳ vọng của `PauliZ` trên mỗi qubit, tạo đầu ra 4 chiều.
    -   **Tầng hậu xử lý (post_net)**: Một tầng fully connected (Linear) ánh xạ từ 4 chiều sang 10 chiều, dự đoán xác suất cho 10 lớp.
-   **Dữ liệu**: Tập CIFAR-10 được chia thành 70% train và 30% test (từ tập train chính thức). Dữ liệu được chuẩn hóa bằng `transforms.Normalize`.
-   **Huấn luyện**:
    -   Sử dụng loss function `CrossEntropyLoss`.
    -   Tối ưu hóa bằng `Adam` với learning rate `lr=0.001`.
    -   Mixed precision training (`torch.amp`) để tăng tốc trên GPU.
    -   Early stopping với patience=3, lưu mô hình tốt nhất dựa trên train loss.
    -   Huấn luyện qua tối đa 10 epoch.
-   **Công cụ**: PennyLane để xây dựng mạch lượng tử, PyTorch cho các tầng cổ điển. Thiết bị mô phỏng là `lightning.gpu`.

### Phương pháp triển khai QNN ban đầu

1. **Tiền xử lý dữ liệu**:

    - Tải và chuẩn hóa dữ liệu CIFAR-10 bằng `torchvision.datasets.CIFAR10`.
    - Chia tập train thành train/test (70/30) bằng `SubsetRandomSampler`.
    - Batch size là 32, sử dụng `DataLoader` để xử lý dữ liệu theo lô.

2. **Xây dựng mô hình**:

    - **Tầng tích chập**: Áp dụng `Conv2d` (3->16 kênh), ReLU, giảm kích thước từ 32x32 xuống 15x15 (stride 2).
    - **Tiền xử lý**: Dữ liệu được reshape (16*15*15=3600 chiều) và ánh xạ qua tầng `pre_net` xuống 4 chiều, phù hợp với 4 qubit.
    - **Mạch lượng tử**:
        - Mã hóa dữ liệu bằng cổng `RX`, ánh xạ từng giá trị đầu vào vào một qubit.
        - Áp dụng 2 tầng mạch, mỗi tầng gồm:
            - Cổng `RY` với tham số học được (`weights`) để tạo ánh xạ phi tuyến.
            - Cổng `CZ` giữa các qubit liền kề để tạo rối lượng tử.
        - Đo giá trị kỳ vọng của `PauliZ` trên 4 qubit, tạo đầu ra 4 chiều.
    - **Hậu xử lý**: Tầng `post_net` gồm một tầng Linear (4->10), dự đoán xác suất cho 10 lớp.

3. **Huấn luyện**:
    - Mô hình được huấn luyện qua tối đa 10 epoch, sử dụng `Adam` để tối ưu hóa cả tham số cổ điển (conv, pre_net, post_net) và tham số lượng tử (weights trong mạch).
    - Loss được tính bằng `CrossEntropyLoss`, tối ưu hóa bằng gradient descent với mixed precision.
    - Đánh giá định kỳ trên tập test, lưu mô hình nếu train loss cải thiện.
    - Early stopping với patience=3 để tránh huấn luyện quá lâu nếu mô hình không tiến bộ.

## Kết quả (QNN ban đầu)

-   **Epoch 1**: Train Loss: 2.3162, Train Acc: 9.98%, Test Loss: 2.3034, Test Acc: 9.67%
-   **Epoch 5**: Train Loss: 2.3036, Train Acc: 10.00%, Test Loss: 2.3034, Test Acc: 10.08%
-   **Epoch 6**: Train Loss: 2.3036, Train Acc: 9.36%, Test Loss: 2.3034, Test Acc: 9.90%
-   Early stopping sau 6 epoch (loss không cải thiện).
-   **Phân tích**: Độ chính xác ~10% (như ngẫu nhiên), loss không giảm, mô hình không học được.

## Thiếu sót và hạn chế (QNN ban đầu)

1. **Hiệu suất thấp**:
    - Tầng convolution (3→16 kênh, stride 2) giảm kích thước mạnh, có thể làm mất thông tin quan trọng.
    - 4 qubit và mạch lượng tử đơn giản không đủ mã hóa dữ liệu phức tạp.
2. **Dữ liệu**: Tập test lấy từ tập train, không khách quan.
3. **Tính toán**: Xử lý từng mẫu trong batch qua mạch lượng tử, rất chậm.

## Đề xuất cải tiến (QNN ban đầu)

1. Tăng số tầng convolution (hoặc dùng ResNet) để trích xuất đặc trưng tốt hơn trước khi đưa vào QNN.
2. Tăng số qubit (8-10) và độ phức tạp mạch (thêm layer, dùng `StronglyEntanglingLayers`).
3. Sử dụng tập test độc lập của CIFAR-10.
4. Vector hóa xử lý batch trong mạch lượng tử để tăng tốc.
