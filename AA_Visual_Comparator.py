import cv2
import numpy as np
from PIL import Image


# --- توابع Anti-aliasing بدون تغییر ---
def aa_gaussian_blur(img_cv, kernel_size=(7, 7)):
    """اعمال Anti-aliasing با فیلتر گوسی."""
    return cv2.GaussianBlur(img_cv, kernel_size, 0)


def aa_median_blur(img_cv, kernel_size=7):
    """اعمال Anti-aliasing با فیلتر میانه."""
    return cv2.medianBlur(img_cv, kernel_size)


def aa_resample(img_pil, scale_factor=2):
    """اعمال Anti-aliasing با شبیه‌سازی Super-Sampling (SSAA) با استفاده از PIL."""
    original_size = img_pil.size
    super_sampled_size = (original_size[0] * scale_factor, original_size[1] * scale_factor)
    temp_large_img = img_pil.resize(super_sampled_size, Image.Resampling.BICUBIC)
    anti_aliased_img_pil = temp_large_img.resize(original_size, Image.Resampling.LANCZOS)
    return cv2.cvtColor(np.array(anti_aliased_img_pil), cv2.COLOR_RGB2BGR)


# --- تابع کمکی: تغییر مقیاس و برچسب‌گذاری ---
def resize_and_label(image, label_text, target_size, color=(255, 255, 255)):
    """تغییر اندازه تصویر به target_size و افزودن برچسب."""
    # تغییر اندازه تصویر به ابعاد استاندارد با استفاده از روش INTER_AREA (مناسب برای کوچک‌سازی)
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # تعریف پارامترهای بصری برای برچسب‌گذاری
    font_scale = 0.8
    font_thickness = 2

    # افزودن متن (برچسب) به بالای تصویر
    labeled_image = cv2.putText(
        resized_image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, color, font_thickness
    )
    return labeled_image


# --- تابع اصلی: اجرا و نمایش مقایسه‌ای (با تغییرات) ---
def run_all_anti_aliasing_methods(image_path):
    # 1. تنظیم ابعاد استاندارد برای نمایش
    # ⚠️ ابعاد استاندارد نمایش خروجی را اینجا تنظیم کنید.
    STANDARD_WIDTH = 500
    STANDARD_HEIGHT = 300
    TARGET_SIZE = (STANDARD_WIDTH, STANDARD_HEIGHT)

    # 2. خواندن تصویر و اعتبارسنجی
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"خطا: فایل تصویر در مسیر {image_path} برای PIL پیدا نشد.")
        return

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"خطا: نتوانستم تصویر را در مسیر {image_path} با OpenCV بخوانم.")
        return

    # 3. اجرای روش‌ها

    # الف. تصویر اصلی
    original = img_cv.copy()

    # ب. فیلتر گوسی
    aa_gaussian = aa_gaussian_blur(img_cv.copy())

    # ج. فیلتر میانه
    aa_median = aa_median_blur(img_cv.copy())

    # د. Super-Sampling
    aa_resample_output = aa_resample(img_pil, scale_factor=2)

    # 4. تغییر مقیاس، برچسب‌گذاری و ترکیب

    # تصاویر برچسب‌گذاری شده و هم‌اندازه
    original_labeled = resize_and_label(original, "1. Original Image (Aliased)", TARGET_SIZE, (0, 0, 255))
    aa_gaussian_labeled = resize_and_label(aa_gaussian, "2. Gaussian Blur (Spatial AA)", TARGET_SIZE, (0, 255, 0))
    aa_median_labeled = resize_and_label(aa_median, "3. Median Blur (Edge-Preserving AA)", TARGET_SIZE, (255, 0, 0))
    aa_resample_labeled = resize_and_label(aa_resample_output, "4. Resampling/SSAA (Highest Quality)", TARGET_SIZE,
                                           (255, 255, 0))

    # ترکیب خروجی‌ها در دو ردیف
    top_row = np.hstack((original_labeled, aa_gaussian_labeled))
    bottom_row = np.hstack((aa_median_labeled, aa_resample_labeled))

    comparison_image = np.vstack((top_row, bottom_row))

    # نمایش تصویر مقایسه‌ای
    cv2.imshow("Anti-aliasing Methods Comparison", comparison_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # ⚠️ مسیر فایل تصویر خود را در متغیر زیر وارد کنید.
#IMAGE_FILE_PATH = r""

run_all_anti_aliasing_methods(IMAGE_FILE_PATH)