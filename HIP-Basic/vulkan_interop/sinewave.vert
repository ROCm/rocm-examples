#version 450

layout(location = 0) in float height;
layout(location = 1) in vec2 xy;

layout(location = 0) out float frag_height;

void main()
{
    gl_Position = vec4(xy, 0, 1);
    frag_height = height;
}
