
FROM python:3.9-slim

# WAJIB di Hugging Face: Buat user non-root agar mendapat izin menulis file
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"


WORKDIR /app


COPY --chown=user . /app


RUN pip install --no-cache-dir -r requirements.txt

# Buka port 7860 
EXPOSE 7860


CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
