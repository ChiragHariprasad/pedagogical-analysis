FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . /code

ENV PORT=7860
EXPOSE 7860

CMD ["python", "main.py"]
