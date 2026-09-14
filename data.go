package microgpt

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
)

const FILENAME = "input.txt"
const URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"

func downloadDataset(filename, url string) {
	out, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer out.Close()
	resp, err := http.Get(url)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		panic(fmt.Errorf("bad status: %s", resp.Status))
	}

	if _, err := io.Copy(out, resp.Body); err != nil {
		panic(err)
	}
}

func getDataset(filename, url string) []string {
	if _, err := os.Stat(FILENAME); errors.Is(err, os.ErrNotExist) {
		downloadDataset(filename, url)
	}

	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	dataset := []string{}
	for scanner.Scan() {
		dataset = append(dataset, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		panic(err)
	}

	return dataset
}
