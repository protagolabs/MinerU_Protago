"marker_text": 
<html><body><table><tbody><tr><th>联系人和联系方式</th><th>董事会秘书</th><th>证券事务代表</th></tr><tr><td>姓名</td><td>王艳辉</td><td>王宁</td></tr><tr><td>办公地址</td><td>山东省烟台市牟平区新城大街889号</td><td>山东省烟台市牟平区新城<br>大街889号</td></tr><tr><td>电话</td><td>0535-3396069</td><td>0535-3396069</td></tr><tr><td>电子信箱</td><td>andrezq@northandre.com</td><td>andrezq@northandre.com</td></tr></tbody></table></body></html>
"azure_text": 
<html><body><table>\n\t<tr>\n\t\t<td colspan=\"1\" rowspan=\"1\">联系人和联系方式</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">董事会秘书</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">证券事务代表</td>\n\t</tr>\n\t<tr>\n\t\t<td colspan=\"1\" rowspan=\"1\">姓名</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">王艳辉</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">王宁</td>\n\t</tr>\n\t<tr>\n\t\t<td colspan=\"1\" rowspan=\"1\">办公地址</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">山东省烟台市牟平区新城大街889号</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">山东省烟台市牟平区新城 大街889号</td>\n\t</tr>\n\t<tr>\n\t\t<td colspan=\"1\" rowspan=\"1\">电话</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">0535-3396069</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">0535-3396069</td>\n\t</tr>\n\t<tr>\n\t\t<td colspan=\"1\" rowspan=\"1\">电子信箱</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">andrezq@northandre. com</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">andrezq@northandre. com</td>\n\t</tr>\n</table></body></html>

"similarity_score": 0.8094445600166424,
"structure_similarity_score": 0.8181818181818181, 

"mineru_text": 
<html><body><table><tr><td>联系人和联系方式</td><td>董事会秘书</td><td>证券事务代表</td></tr><tr><td>姓名</td><td>王艳辉</td><td>王宁</td></tr><tr><td>办公地址</td><td>山东省烟台市牟平区新城大街889号</td><td>山东省烟台市牟平区新城 大街889号</td></tr><tr><td>电话</td><td>0535-3396069</td><td>0535-3396069</td></tr><tr><td>电子信箱</td><td> andrezq@northandre.com</td><td>andrezq@northandre.com</td></tr></table></body></html>
"azure_text": 
<html><body><table>\n\t<tr>\n\t\t<td colspan=\"1\" rowspan=\"1\">联系人和联系方式</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">董事会秘书</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">证券事务代表</td>\n\t</tr>\n\t<tr>\n\t\t<td colspan=\"1\" rowspan=\"1\">姓名</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">王艳辉</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">王宁</td>\n\t</tr>\n\t<tr>\n\t\t<td colspan=\"1\" rowspan=\"1\">办公地址</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">山东省烟台市牟平区新城大街889号</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">山东省烟台市牟平区新城 大街889号</td>\n\t</tr>\n\t<tr>\n\t\t<td colspan=\"1\" rowspan=\"1\">电话</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">0535-3396069</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">0535-3396069</td>\n\t</tr>\n\t<tr>\n\t\t<td colspan=\"1\" rowspan=\"1\">电子信箱</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">andrezq@northandre. com</td>\n\t\t<td colspan=\"1\" rowspan=\"1\">andrezq@northandre. com</td>\n\t</tr>\n</table></body></html>

"similarity_score": 0.9934782608695653,
"structure_similarity_score": 1.0,

Q: visually, these texts show same tables, why the TED scores are much different.

A: You make an excellent point! Visually, these three HTML tables would render exactly the same to a human reader - they all show the same contact information table with the same layout and content. This highlights an important limitation of the TEDS metric:

TEDS is very sensitive to HTML structure differences, even when they don't affect visual appearance. Let's break down why:

1. Looking at the TEDS implementation in `metric.py`, it treats structural differences quite strictly:

```12:15:utils/metric.py
def rename(self, node1, node2):
    """Compares attributes of trees"""
    if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
        return 1.
```

This code shows that TEDS will give a full penalty (1.0) for:
- Different tags (`th` vs `td`) even though they might render similarly
- Different attributes even if they're functionally equivalent (like missing `colspan="1"` vs having it explicitly)
- Structural elements like `<tbody>` that are optional and don't affect rendering

2. The metric doesn't account for HTML's forgiving nature:
- `<th>` and `<td>` both create table cells (just with different default styling)
- `colspan="1"` is the same as not having colspan specified
- Extra whitespace is collapsed in HTML rendering
- `<tbody>` is optional and automatically added by browsers
- `<br>` vs space character might render similarly in some contexts

This suggests that TEDS might not be the best metric if your goal is to measure visual or functional similarity of tables. You might want to consider:

1. Using a more visually-oriented metric that compares rendered output
2. Preprocessing the HTML to normalize it before comparison (e.g., removing optional attributes, normalizing tags)
3. Using a modified version of TEDS that weights structural differences less heavily
4. Adding visual similarity as an additional evaluation component

