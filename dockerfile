FROM jupyter/base-notebook

WORKDIR /home/jovyan/eos-widget
COPY . /home/jovyan/eos-widget


RUN pip install voila
RUN pip install https://github.com/t-reents/acwf-verification-scripts/archive/refs/heads/imp/gen_periodic_tables.zip
RUN pip install --no-cache-dir -e .

RUN  python -m ipykernel install --user

EXPOSE 8866

CMD ["voila", "eos-widget.ipynb", "--port=8866", "--no-browser", "--Voila.ip=0.0.0.0"]