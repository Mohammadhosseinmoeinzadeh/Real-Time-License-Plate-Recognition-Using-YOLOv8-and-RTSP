# Real-Time-License-Plate-Recognition-Using-YOLOv8-and-RTSP


# پلاک‌خوان خودکار خودرو (ALPR) با استفاده از YOLOv8

این پروژه شامل دو بخش آموزش مدل تشخیص پلاک خودرو با استفاده از YOLOv8 و استفاده از مدل آموزش‌دیده برای شناسایی پلاک خودرو در تصاویر زنده (RTSP) است.

---

## ویژگی‌ها

- آموزش مدل YOLOv8 روی دیتاست پلاک ایرانی
- رابط گرافیکی (GUI) برای خواندن پلاک از دوربین‌های RTSP
- نمایش زنده تصویر دوربین و آخرین پلاک شناسایی‌شده
- ذخیره خودکار تصویر پلاک در مسیر دلخواه
- پشتیبانی از کدک H.265 با استفاده از PyAV

---

## آموزش مدل

اسکریپت `yol.py` برای آموزش مدل استفاده می‌شود. این اسکریپت از مدل پایه YOLOv8n استفاده کرده و آن را روی دیتاست پلاک خودرو ایرانی آموزش می‌دهد.

**ساختار فایل `data.yaml` برای آموزش:**

```yaml
path: <مسیر اصلی دیتاست>
train: <زیرپوشه آموزش>
val: <زیرپوشه اعتبارسنجی>
nc: 1
names: ['plate']
```

> ✅ دیتاست مورد استفاده از مخزن [roozbehrajabi/ALPR_Dataset](https://github.com/roozbehrajabi/ALPR_Dataset) گرفته شده است.  
> از زحمات ایشان بابت فراهم‌سازی این دیتاست ارزشمند سپاس‌گزاریم.

---

## اجرای برنامه تشخیص پلاک

اسکریپت `cam4.py` یک رابط گرافیکی (GUI) در محیط Tkinter است که با استفاده از مدل آموزش‌دیده، پلاک را در تصاویر دریافتی از دوربین RTSP شناسایی کرده و ذخیره می‌کند.

**ویژگی‌های رابط گرافیکی:**

- امکان وارد کردن آدرس RTSP
- نمایش تصویر زنده دوربین
- نمایش آخرین پلاک ذخیره‌شده
- انتخاب مسیر ذخیره برای تصاویر
- استفاده از مدل YOLOv8 برای تشخیص

---

## پیش‌نیازها

- Python 3.8+
- کتابخانه‌ها:
  - `ultralytics`
  - `opencv-python`
  - `Pillow`
  - `av`
  - `tkinter` (در بیشتر نصب‌های Python به صورت پیش‌فرض موجود است)

نصب پیش‌نیازها:

```bash
pip install ultralytics opencv-python pillow av
```

---

## آموزش مدل

```bash
python yol.py
```

> 🔧 حتماً آدرس‌های مربوط به `data.yaml` را طبق مسیر دیتاست خود ویرایش کنید.

---

## اجرای برنامه RTSP

در فایل `cam4.py`، مسیر مدل آموزش‌دیده خود را وارد کنید:

```python
model_path = "<YOUR_MODEL_ADDRESS>"
```

سپس اجرا کنید:

```bash
python cam4.py
```
---
## تصویر محیط برنامه 
<div align="center">
  <img src="https://github.com/Mohammadhosseinmoeinzadeh/Real-Time-License-Plate-Recognition-Using-YOLOv8-and-RTSP/blob/main/app.png" width="600"/>
</div>

---
---

## نمونه تصویر خروجی

<div align="center">
  <img src="https://github.com/Mohammadhosseinmoeinzadeh/Real-Time-License-Plate-Recognition-Using-YOLOv8-and-RTSP/blob/main/plate_1745478787.jpg" width="600"/>
</div>

---

## مجوز

این پروژه برای اهداف آموزشی ارائه شده است.
