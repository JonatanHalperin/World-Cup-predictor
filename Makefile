.PHONY: build-data clean-data

build-data:
	bash scripts/build_data.sh $(ARGS)

clean-data:
	bash scripts/clean_data.sh $(ARGS)
