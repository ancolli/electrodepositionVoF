#!/bin/bash

output="resultados.txt"
echo -e "Tiempo\tPosicion" > "$output"

for dir in [0-9]*; do
    if [ -d "$dir" ] && [[ "$dir" =~ ^[0-9]+$ ]]; then
        file="$dir/interfaceCentre.water"
        if [ -f "$file" ]; then
            y_val=$(awk '
                /internalField[ \t]+nonuniform[ \t]+List<vector>/ {inField=1; next}
                inField==1 && /^[0-9]+$/ {inField=2; next}   # skip number of vectors
                inField==2 && /^\(/ {inField=3; next}        # opening bracket
                inField==3 && /^\)/ {exit}                  # closing bracket
                inField==3 {
                    gsub(/\(|\)/,"",$0)
                    split($0,a," ")
                    if (a[1]!=0 && a[2]!=0 && a[3]!=0) {
                        print a[2]
                        exit
                    }
                }
            ' "$file")
            
            if [ ! -z "$y_val" ]; then
                echo -e "${dir}\t${y_val}" >> "$output"
            fi
        fi
    fi
done

sort -n "$output" -o "$output"
echo "Hecho. Resultados en $output"





