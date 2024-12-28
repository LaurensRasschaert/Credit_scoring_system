FROM python:3.9-slim

#Workdirectory
WORKDIR /app

#copy files to container
COPY . /app

# install the reqs
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8000

# start app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
