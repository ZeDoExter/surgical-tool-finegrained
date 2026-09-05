# Progress UI Specification

## Goal

ปรับ console progress ตอน training ให้เป็นสไตล์ **YOLO-like** ที่ดู clean, compact และอ่านง่าย

ไม่ต้องทำ dashboard หรือ Rich UI

ต้องการเพียง:

- Live progress bar ระหว่าง batch
- Epoch summary หลังจบแต่ละ epoch
- แสดงเวลาในการ train แต่ละ epoch
- แสดง ETA ของ training ที่เหลือ
- แสดง metrics สำคัญหลังจบ epoch
- แสดง `*best*` เมื่อ epoch นั้นทำ best metric ใหม่
- ไม่ให้ console รก

---

## 1. Batch Progress Bar

ใช้ `tqdm` เป็น progress bar หลัก

รูปแบบที่ต้องการ:

```text
1/60 ━━━━━━━━━━━━━━━━━━━━━━━━ 42% 84/200 loss=0.684 ce=0.421 dice=0.263
```

เมื่อจบ epoch:

```text
1/60 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 200/200 loss=0.654 ce=0.412 dice=0.242
```

### Configuration

ใช้ประมาณนี้:

```python
pbar = tqdm(
    dl,
    desc=f"{epoch}/{cfg.epochs}",
    unit="batch",
    bar_format="{l_bar}{bar:24}{r_bar}",
    leave=False,
    dynamic_ncols=True,
    mininterval=1.0,
)
```

### Live metrics

ระหว่าง batch ให้แสดงเฉพาะ:

```text
loss
ce
dice
```

ตัวอย่าง:

```text
loss=0.654 ce=0.412 dice=0.242
```

ใช้:

```python
pbar.set_postfix(
    loss=f"{loss.item():.3f}",
    ce=f"{parts['ce']:.3f}",
    dice=f"{parts['dice']:.3f}",
)
```

ห้ามใส่ metrics ที่ยังไม่มีระหว่าง training เช่น:

- Precision
- Recall
- F1
- IoU

---

# 2. Epoch Timing

เมื่อเริ่ม epoch ให้บันทึกเวลา:

```python
epoch_start = time.time()
```

เมื่อ epoch เสร็จ:

```python
epoch_time = time.time() - epoch_start
```

แสดงเวลาที่ใช้ใน epoch นั้น

ตัวอย่าง:

```text
48.3s
```

ถ้าเกิน 1 นาที:

```text
1m 12s
```

---

# 3. Time Formatter

เพิ่ม helper สำหรับ format เวลา:

```python
def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))

    if seconds < 60:
        return f"{seconds}s"

    minutes, sec = divmod(seconds, 60)

    if minutes < 60:
        return f"{minutes}m {sec:02d}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"
```

---

# 4. ETA

หลังจบแต่ละ epoch ให้คำนวณ ETA จากเวลาเฉลี่ยต่อ epoch

```python
elapsed_total = time.time() - t_train0
avg_epoch_time = elapsed_total / epoch
eta = avg_epoch_time * (cfg.epochs - epoch)
```

แสดงเป็น:

```text
ETA=47m 12s
```

ตัวอย่าง:

```text
Epoch 1/60 ... ETA=47m 12s
Epoch 2/60 ... ETA=45m 31s
Epoch 3/60 ... ETA=44m 02s
```

Epoch สุดท้าย:

```text
ETA=0s
```

---

# 5. Epoch Summary

หลัง training + validation ของแต่ละ epoch เสร็จ ให้พิมพ์ **หนึ่งบรรทัดถาวร**

รูปแบบหลัก:

```text
Epoch 1/60 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.3s | loss=0.6541 P=0.842 R=0.786 F1=0.813 IoU=0.721 ETA=47m 12s *best*
```

สำหรับ epoch ที่ไม่ใช่ best:

```text
Epoch 2/60 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.9s | loss=0.5872 P=0.851 R=0.809 F1=0.829 IoU=0.738 ETA=45m 31s
```

---

# 6. Metrics ใน Epoch Summary

แสดง:

```text
loss
P
R
F1
IoU
```

Format:

```text
loss=0.6541
P=0.842
R=0.786
F1=0.813
IoU=0.721
```

ไม่ต้องแสดง:

- confusion matrix
- จำนวน TP
- จำนวน GT
- จำนวน detections
- รายละเอียด validation อื่น ๆ

---

# 7. Best Indicator

ถ้า epoch ปัจจุบันทำ best metric ใหม่ ให้ต่อท้าย:

```text
*best*
```

ตัวอย่าง:

```text
Epoch 4/60 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 47.1s | loss=0.4912 P=0.871 R=0.844 F1=0.857 IoU=0.779 ETA=43m 02s *best*
```

ถ้าไม่ใช่ best:

```text
Epoch 5/60 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.8s | loss=0.4721 P=0.868 R=0.839 F1=0.853 IoU=0.775 ETA=42m 15s
```

ห้ามเปลี่ยน logic ที่ใช้ตัดสิน `best`

ให้ใช้ตัวแปร/logic เดิมของ training code

---

# 8. Final Appearance

ต้องการ output โดยรวมประมาณนี้:

```text
[train] steps/epoch=200 epochs=60

1/60 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 200/200 loss=0.654 ce=0.412 dice=0.242
Epoch 1/60 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.3s | loss=0.6541 P=0.842 R=0.786 F1=0.813 IoU=0.721 ETA=47m 12s *best*

2/60 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 200/200 loss=0.587 ce=0.381 dice=0.206
Epoch 2/60 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.9s | loss=0.5872 P=0.851 R=0.809 F1=0.829 IoU=0.738 ETA=45m 31s *best*

3/60 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 200/200 loss=0.541 ce=0.352 dice=0.189
Epoch 3/60 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.4s | loss=0.5410 P=0.847 R=0.812 F1=0.829 IoU=0.741 ETA=44m 02s
```

---

# 9. Console Cleanliness

ต้องการให้ progress bar เป็น dynamic line ของ `tqdm`

ไม่ต้อง print ทุก batch ด้วย `print()` เอง

ไม่ต้องสร้าง progress bar ซ้อนกัน

ไม่ต้องใช้:

- Rich
- Textual
- custom terminal escape codes
- dashboard
- panels
- tables
- colors
- emojis

ใช้ `tqdm` ธรรมดา แต่จัด format ให้ดูเหมือน YOLO

---

# 10. Important

นี่เป็น **Progress UI change เท่านั้น**

ห้ามเปลี่ยน:

- training logic
- model
- loss
- optimizer
- scheduler
- dataset
- dataloader
- validation
- checkpoint
- best-model logic
- early stopping
- AMP
- batch size
- learning rate

เปลี่ยนเฉพาะ:

1. `tqdm` appearance
2. epoch timer
3. ETA
4. epoch summary output
5. `format_time()` helper

ผลลัพธ์ควรให้ความรู้สึกเหมือน:

```text
YOLO training output
+ clean epoch summary
+ per-epoch timing
+ ETA
```

โดยเน้น **เรียบ, สั้น, อ่านง่าย และไม่รก**