CC = gcc
HCFLAGS = -std=c99 -O2 -g
SCFLAGS = -std=c99 -O2 -g
LDFLAGS = -lm
HEADER_FILES = function.h args.h util.h

BUILD = ./obj

main : $(BUILD)/local_main.o $(BUILD)/function.o $(BUILD)/master.o
	$(CC) $(HCFLAGS) -o $@ $^ $(LDFLAGS) -lpthread

$(BUILD)/%.o : %.c $(HEADER_FILES)
	$(CC) $(HCFLAGS) -o $@ -c $< 

$(BUILD)/slave.o : slave.c args.h
	$(CC) $(SCFLAGS) -o $@ -c $<

run : 
	./main

clean :
	rm -f $(BUILD)/*.o ./main
