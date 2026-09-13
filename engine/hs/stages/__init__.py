"""Stage implementations. Each exposes ``add_parser(sub)`` and ``run(args, project)``;
cli.py wires them. Everything that computes is a subprocess of a vendored script or a
binary — see runner.py."""
