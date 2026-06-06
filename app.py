from mythograph.ui.blocks import build_demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
