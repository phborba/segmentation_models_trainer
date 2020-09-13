FROM tensorflow/tensorflow:latest-gpu-jupyter
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt && \
    pip install -U git+https://github.com/phborba/efficientnet && \
    pip install -U git+https://github.com/phborba/segmentation_models
CMD ["bash" "-c" "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/github_repos --ip 0.0.0.0 --no-browser --allow-root"]