CC = gcc
HCFLAGS = -std=c99 -O3
SCFLAGS = -std=c99 -O3
LDFLAGS = -lm
HEADER_FILES = function.h args.h util.h

BUILD = ./obj

main : $(BUILD)/local_main.o $(BUILD)/function.o
	$(CC) $(HCFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/%.o : %.c $(HEADER_FILES)
	$(CC) $(HCFLAGS) -o $@ -c $< 

$(BUILD)/slave.o : slave.c args.h
	$(CC) $(SCFLAGS) -o $@ -c $<

run : 
	./main

clean :
	rm -f $(BUILD)/*.o ./main
