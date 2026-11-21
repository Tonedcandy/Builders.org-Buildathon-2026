FROM rust:1.83-bookworm AS builder
WORKDIR /build
ENV RUSTFLAGS="-C target-cpu=x86-64-v2"
COPY Cargo.toml ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release && rm -rf src
COPY src ./src
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim
RUN useradd --uid 1000 --no-create-home --home-dir /scratch runner
COPY --from=builder /build/target/release/shopping-sessions /app/shopping-sessions
RUN chmod +x /app/shopping-sessions
WORKDIR /app
USER 1000:1000
ENTRYPOINT ["/app/shopping-sessions"]