import yt_dlp

url = "https://youtu.be/62DxELjuRec"

ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'outtmpl': '%(title)s.%(ext)s',
    'merge_output_format': 'mp4',  # Force final file format
    'postprocessors': [{
        'key': 'FFmpegVideoConvertor',
        'preferedformat': 'mp4',  # Ensure mp4 output
    }],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
