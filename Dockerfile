# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code (including templates folder)
COPY . .

# <<< تغییر کلیدی: یک پوشه قابل نوشتن برای فایل‌های موقت ایجاد می‌کنیم >>>
# این پوشه توسط اپلیکیشن برای دانلود و آپلود فایل‌ها استفاده خواهد شد.
RUN mkdir -p /app/tmp && chmod 777 /app/tmp

# Expose the port the app runs on
EXPOSE 7860

# <<< تغییر کلیدی: بهینه سازی برای پردازش همزمان کاربران بیشتر >>>
# Run app.py when the container launches
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--bind", "0.0.0.0:7860", "app:app"]
