## Cash Labeling Script

Run the following command in the terminal

```
python cashLabeling.py "dataset/videos/greenscreen_50.MOV" "dataset/image/RM50/" True 0
```

#### The arguments for the script
```
python cashLabeling.py <video_path> <output_path> <collect_data> <verbose>
```
- `video_path` : the path to the video file
- `output_path` : the path the output file
- `collect_data` : flag to indicate whether to collect the data (`True` or `False`)
- `verbose` :
  - `1` - File name and bounding box
  - `2` - Video output


## File Renaming Script

Run the following command in the terminal

```
rm dataset/image/RM50/.DS_Store
python renamingScript.py "dataset/image/rm50/" ".png" 0
```

#### The arguments for the script
```
python cashLabeling.py  <folder_path> <file_type> <resume_from>
```
- `folder_path` : the path to the folder where the renaming is done
- `file_type` : the file type inside the folder
- `resume_from` : the number to resume when renaming the data (-1 to start from 0)
