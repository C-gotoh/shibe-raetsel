default: run

clean:
	@[[ ! -e shibe-raetsel.love ]] || rm shibe-raetsel.love
	@[[ ! -e pkg ]] || rm -r pkg        

build: clean
	@zip -r -0 shibe-raetsel.love data/*
	@zip -r -0 shibe-raetsel.love lib
	@cd src/ && zip -r ../shibe-raetsel.love *

run: build
	@love-hg shibe-raetsel.love