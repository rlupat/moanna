FROM python:3.8.2
LABEL maintainer="Richard Lupat <richard.lupat@petermac.org>"

WORKDIR /moanna
COPY . .

# Install packages
RUN pip3 install -r requirements.txt

# PATH
ENV PATH="/moanna/bin/:${PATH}"