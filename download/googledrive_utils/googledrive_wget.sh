wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12345678' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=12345678" -O 12345678.zip && rm -rf /tmp/cookies.txt