buildCommand = python setup.py build_ext --build-lib=../ --build-temp=cy/temp
cleanCommand = rm cy/*.so; rm -rf cy/temp


all: #pivot
		$(buildCommand)

clean:
		$(cleanCommand)

cleanall:
		$(cleanCommand)
		rm cy/Cy*.cpp  # So rarely necessary

re:
		$(cleanCommand)
		$(buildCommand)


