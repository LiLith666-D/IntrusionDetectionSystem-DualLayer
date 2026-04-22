FROM nginx:1.27-alpine

# Remove default nginx page
RUN rm /etc/nginx/conf.d/default.conf

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY dashboard.html /usr/share/nginx/html/index.html

EXPOSE 80
