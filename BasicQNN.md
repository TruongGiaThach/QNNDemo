# Báo cáo phân tích giải pháp QNN mới trên CIFAR-10

## 1. Mô tả giải pháp

Giải pháp mới triển khai một **Quantum Neural Network (QNN)** lai (hybrid quantum-classical) cải tiến để phân loại hình ảnh trên tập dữ liệu CIFAR-10 (10 lớp, hình ảnh 32x32x3). Các thành phần chính của giải pháp bao gồm:

- **Kiến trúc mạng**:
  - **Tầng tích chập cổ điển**: Hai tầng `Conv2d` (3->32 kênh, 32->64 kênh, kernel 3x3, padding=1) kết hợp với `MaxPool2d` (2x2) để trích xuất đặc trưng không gian, giảm kích thước ảnh từ 32x32 xuống 8x8.
  - **Tầng tiền xử lý (pre_net)**: Tầng fully connected (Linear) ánh xạ từ 64*8*8 (4096 chiều) xuống 16 chiều (2*n_qubits, với n_qubits=8), cung cấp đầu vào cho mạch lượng tử.
  - **Mạch lượng tử**: Mạch lượng tử sử dụng 8 qubit, 6 tầng (n_layers=6), với:
    - Mã hóa dữ liệu bằng `AmplitudeEmbedding` để tận dụng không gian Hilbert.
    - Cấu trúc **strongly entangling layers** với cổng `CRX`, `CRY` (tham số học được), và `CNOT` (rối lượng tử).
    - Đo giá trị kỳ vọng của `PauliZ` và `PauliX` trên mỗi qubit, tạo đầu ra 16 chiều.
  - **Tầng hậu xử lý (post_net)**: Một mạng fully connected với hai tầng Linear (16->128, 128->10) và ReLU để học ánh xạ phi tuyến từ đầu ra của mạch lượng tử sang 10 lớp.
- **Dữ liệu**: Tập CIFAR-10 được chia thành 70% train và 30% test (từ tập train chính thức). Dữ liệu được chuẩn hóa bằng `transforms.Normalize`.
- **Huấn luyện**:
  - Sử dụng loss function `CrossEntropyLoss`.
  - Tối ưu hóa bằng `Adam` với learning rate `lr=0.0005`.
  - Áp dụng scheduler `ReduceLROnPlateau` để điều chỉnh learning rate dựa trên test loss.
  - Mixed precision training (`torch.amp`) để tăng tốc trên GPU.
  - Early stopping với patience=5, lưu mô hình tốt nhất dựa trên test loss.
  - Huấn luyện qua tối đa 50 epoch.
- **Công cụ**: PennyLane để xây dựng mạch lượng tử, PyTorch cho các tầng cổ điển. Thiết bị mô phỏng là `default.qubit` (có thể chuyển sang `lightning.gpu` hoặc phần cứng NISQ như IBM Quantum).

## 2. Kết quả (dự kiến)

- **Độ chính xác**: Dự kiến độ chính xác trên tập test đạt khoảng **40-50%** sau 50 epoch, cải thiện đáng kể so với giải pháp cũ (~10%). Mức này phù hợp với QNN trên các thiết bị NISQ cho CIFAR-10, một tập dữ liệu phức tạp.
- **Loss**: Loss trên tập train và test giảm ổn định qua các epoch, cho thấy mô hình học được các đặc trưng hữu ích hơn so với giải pháp cũ.
- **Hiệu suất**:
  - Thời gian huấn luyện mỗi epoch lâu hơn do số qubit tăng (8 qubit), độ sâu mạch lớn (6 tầng), và kiến trúc mạng phức tạp hơn.
  - Mô phỏng trên CPU/GPU có thể mất vài giờ cho 50 epoch, nhưng khả thi với `lightning.gpu` hoặc phần cứng lượng tử thực tế.
- **Cải thiện**: Độ chính xác tăng nhờ mạch lượng tử biểu cảm hơn, kiến trúc CNN mạnh hơn, và huấn luyện tối ưu hơn.

## 3. Phương pháp

Phương pháp triển khai QNN mới bao gồm các bước sau:

1. **Tiền xử lý dữ liệu**:
   - Tải và chuẩn hóa dữ liệu CIFAR-10 bằng `torchvision.datasets.CIFAR10`.
   - Chia tập train thành train/test (70/30) bằng `SubsetRandomSampler`.
   - Batch size tăng lên 64 để ổn định gradient, sử dụng `DataLoader` để xử lý dữ liệu theo lô.

2. **Xây dựng mô hình**:
   - **Tầng tích chập**:
     - Tầng `conv1`: Áp dụng `Conv2d` (3->32 kênh), ReLU, và `MaxPool2d` để giảm kích thước từ 32x32 xuống 16x16.
     - Tầng `conv2`: Áp dụng `Conv2d` (32->64 kênh), ReLU, và `MaxPool2d` để giảm kích thước xuống 8x8.
   - **Tiền xử lý**: Dữ liệu được reshape (64*8*8=4096 chiều) và ánh xạ qua tầng `pre_net` xuống 16 chiều, phù hợp với `AmplitudeEmbedding` cho 8 qubit.
   - **Mạch lượng tử**:
     - Mã hóa dữ liệu bằng `AmplitudeEmbedding`, ánh xạ vector 16 chiều vào trạng thái lượng tử của 8 qubit (2^8=256 trạng thái).
     - Áp dụng 6 tầng mạch, mỗi tầng gồm:
       - Cổng `CRX`, `CRY` với tham số học được (`weights`) để tạo ánh xạ phi tuyến.
       - Cổng `CNOT` (bao gồm kết nối vòng) để tạo rối lượng tử mạnh.
     - Đo giá trị kỳ vọng của `PauliZ` và `PauliX` trên 8 qubit, tạo đầu ra 16 chiều.
   - **Hậu xử lý**: Tầng `post_net` gồm hai tầng Linear (16->128, 128->10) với ReLU, dự đoán xác suất cho 10 lớp.

3. **Huấn luyện**:
   - Mô hình được huấn luyện qua tối đa 50 epoch, sử dụng `Adam` để tối ưu hóa cả tham số cổ điển (conv, pre_net, post_net) và tham số lượng tử (weights trong mạch).
   - Loss được tính bằng `CrossEntropyLoss`, tối ưu hóa bằng gradient descent với mixed precision.
   - Scheduler `ReduceLROnPlateau` giảm learning rate khi test loss không cải thiện, đảm bảo hội tụ tốt hơn.
   - Đánh giá định kỳ trên tập test, lưu mô hình nếu test loss cải thiện.
   - Early stopping với patience=5 để tránh huấn luyện quá lâu nếu mô hình không tiến bộ.

## 4. Hạn chế của giải pháp

Mặc dù giải pháp mới cải thiện đáng kể so với giải pháp cũ, vẫn tồn tại một số hạn chế:

1. **Hiệu suất chưa cạnh tranh với CNN cổ điển**:
   - Độ chính xác dự kiến (~40-50%) vẫn thấp hơn nhiều so với các mô hình CNN hiện đại trên CIFAR-10 (thường đạt 80-90% với ResNet hoặc VGG). Điều này do hạn chế của QNN trên thiết bị NISQ và độ phức tạp của CIFAR-10.
   - QNN chưa thể hiện rõ **quantum advantage** trên CIFAR-10, đặc biệt khi số qubit và độ sâu mạch còn hạn chế.

2. **Yêu cầu tài nguyên tính toán cao**:
   - Mạch lượng tử với 8 qubit và 6 tầng đòi hỏi tài nguyên lớn để mô phỏng trên CPU/GPU. Mô phỏng có thể chậm (mất vài giờ cho 50 epoch), đặc biệt nếu không có `lightning.gpu` hoặc phần cứng lượng tử thực tế.
   - Nếu chạy trên thiết bị NISQ (như IBM Quantum), nhiễu (noise) từ lỗi cổng và đo lường có thể làm giảm độ chính xác, đòi hỏi kỹ thuật sửa lỗi hoặc giảm nhiễu.

3. **Mã hóa dữ liệu vẫn có giới hạn**:
   - Dù sử dụng `AmplitudeEmbedding`, việc ánh xạ dữ liệu từ 4096 chiều xuống 16 chiều (pre_net) vẫn gây mất mát thông tin. Các phương pháp mã hóa tiên tiến hơn (như hybrid encoding hoặc kernel-based encoding) có thể cải thiện nhưng phức tạp hơn.
   - Số qubit (8) vẫn nhỏ so với độ phức tạp của CIFAR-10. Tăng số qubit (12-16) có thể cải thiện hiệu suất nhưng vượt quá khả năng mô phỏng hiện tại.

4. **Huấn luyện phức tạp**:
   - QNN dễ gặp vấn đề **Barren Plateaus** (gradient nhỏ khi số qubit và độ sâu mạch tăng), có thể làm chậm hội tụ hoặc gây kẹt ở local minima. Giải pháp mới giảm thiểu điều này bằng mạch strongly entangling và learning rate nhỏ, nhưng vẫn không loại bỏ hoàn toàn.
   - Huấn luyện 50 epoch là tốn thời gian, và việc tối ưu hyperparameter (batch size, learning rate, số tầng) đòi hỏi thử nghiệm thêm (grid search).

5. **Đánh giá chưa đầy đủ**:
   - Việc sử dụng tập test từ tập train (70/30) thay vì tập test chính thức của CIFAR-10 có thể làm đánh giá hiệu suất không chính xác. Cần sử dụng tập test riêng để đo lường khả năng tổng quát hóa.
   - Thiếu các kỹ thuật nâng cao như dropout, weight decay, hoặc transfer learning để cải thiện độ chính xác và tránh overfitting.

## 5. Kết luận

Giải pháp QNN mới triển khai một mô hình lai quantum-classical cải tiến cho CIFAR-10, với mạch lượng tử phức tạp hơn (8 qubit, 6 tầng, amplitude encoding), kiến trúc CNN mạnh hơn (2 tầng tích chập, post_net sâu hơn), và huấn luyện tối ưu hơn (learning rate nhỏ, scheduler, 50 epoch). Dự kiến đạt độ chính xác 40-50%, vượt trội so với giải pháp cũ (~10%). Tuy nhiên, hiệu suất vẫn bị giới hạn bởi tài nguyên tính toán, nhiễu NISQ, và độ phức tạp của CIFAR-10. Để cải thiện thêm, cần tăng số qubit, thử nghiệm mã hóa tiên tiến, tích hợp transfer learning, và sử dụng tập test chính thức của CIFAR-10.