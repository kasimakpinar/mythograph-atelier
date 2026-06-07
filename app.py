from mythograph.ui.blocks import build_demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(ssr_mode=False)
