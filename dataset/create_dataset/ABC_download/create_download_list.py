def generate_abc_download_links(tag: str, chunk_start: int = 0, chunk_end: int = 99, output_file: str = None):
    """
    生成 ABC 数据集的下载链接列表，并写入文本文件（可选）。

    参数:
    - tag: 数据类型标签，如 'meta_v00', 'step_v00', 'obj_v00', 'feat_v00' 等
    - chunk_start: 起始chunk编号（默认0）
    - chunk_end: 结束chunk编号（默认99）
    - output_file: 保存链接的文本文件名；如果为None则不写入，仅返回列表

    返回:
    - links: 所有生成的下载链接（list）
    """
    base_url = "https://cad_models.gitlabpages.inria.fr/abc-dataset/v00/"
    links = []

    for i in range(chunk_start, chunk_end + 1):
        link = f"{base_url}{tag}_chunk_{i:04d}.7z"
        links.append(link)

    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(links))

    return links


if __name__ == "__main__":
    generate_abc_download_links("meta_v00", output_file="meta_links.txt")

