package main

import (
	"bufio"
	"fmt"
	"llm_dev/agent"
	"os"
)

var sss string

func main() {

	model := agent.NewModel("http://192.168.65.2:4000", "sk-1234")
	agent := agent.NewBaseAgent("/root/workspace/llm_dev", *model)
	for {
		reader := bufio.NewScanner(os.Stdin)
		fmt.Print("User Prompt> ")
		reader.Scan() // This will read a line of input from the user
		userprompt := reader.Text()

		agent.NewUserTask(userprompt)
	}
}
