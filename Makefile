build: FORCE
	scarb build

clean: FORCE
	scarb clean

fmt: FORCE
	scarb fmt

test: FORCE
	cairo-test -p .

FORCE:
