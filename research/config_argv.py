"""Put an explicit ml_collections config file before its field overrides."""


def config_parent_first(argv, name="agent"):
    """Preserve values/order except moving the single parent assignment first.

    ml_collections must see ``--agent=path`` before ``--agent.field=value``.
    Drivers using only the default config need no normalization.
    """
    args = list(argv)
    flag = f"--{name}"
    parents = []
    index = 1  # argv[0] is the executable/entry point.
    while index < len(args):
        item = args[index]
        if item == flag:
            if index + 1 == len(args) or args[index + 1].startswith("--"):
                raise ValueError(f"Missing value for {flag}")
            parents.append((index, index + 2))
            index += 2
        elif item.startswith(flag + "="):
            if not item.split("=", 1)[1]:
                raise ValueError(f"Missing value for {flag}")
            parents.append((index, index + 1))
            index += 1
        else:
            index += 1
    if len(parents) > 1:
        raise ValueError(f"Repeated parent assignment for {flag}")
    if not parents:
        return args
    start, end = parents[0]
    return args[:1] + args[start:end] + args[1:start] + args[end:]
