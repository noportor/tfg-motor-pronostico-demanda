# syntax=docker/dockerfile:1.7
#
# Visor del experimento — build multi-etapa.
#
#   deps   : npm install (la capa lenta; solo se invalida si cambia package.json)
#   build  : tsc + vite build — el typecheck es el gate: sin types verdes no hay imagen
#   final  : nginx sirviendo la SPA; salidas*/ se monta en runtime, nunca se hornea
#
# Los datos NUNCA entran a la imagen: el repositorio se monta :ro al correr.

FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm install --no-audit --no-fund

FROM deps AS build
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
RUN npm run build

FROM nginx:1.27-alpine AS final
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
