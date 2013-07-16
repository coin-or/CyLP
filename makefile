buildCommand = python setup.py build_ext --build-lib=./ --build-temp=CyLP/cy/temp
cleanCommand = rm CyLP/cy/*.so; rm -rf CyLP/cy/temp


all: #pivot
		$(buildCommand)

clean:
		$(cleanCommand)

cleanall:
		$(cleanCommand)
		rm CyLP/cy/Cy*.cpp  # So rarely necessary

re:
		$(cleanCommand)
		$(buildCommand)


