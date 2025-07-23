FROM alpine:3.17

WORKDIR /var/www

COPY requirements.txt requirements.txt

RUN mkdir -p /var/www/docs/

RUN mkdir /docs

VOLUME /var/www/docs

COPY _build/html /docs

CMD [ "cp", "-R" , "/docs" , "/var/www" ]
