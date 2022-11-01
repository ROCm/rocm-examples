#version 450

layout(location = 0) out vec4 out_color;

layout(location = 0) in float frag_height;

void main()
{
    out_color = vec4(vec3(frag_height * 0.5 + 0.5), 1.0);
}
