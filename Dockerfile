FROM rust:1.83-alpine AS builder
WORKDIR /build
RUN apk add --no-cache musl-dev linux-headers
ENV RUSTFLAGS="-C target-cpu=x86-64-v2 -C target-feature=+crt-static"
COPY Cargo.toml ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release --target x86_64-unknown-linux-musl && rm -rf src
COPY src ./src
RUN touch src/main.rs && cargo build --release --target x86_64-unknown-linux-musl

FROM scratch
COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/shopping-sessions /shopping-sessions
USER 1000:1000
ENTRYPOINT ["/shopping-sessions"]