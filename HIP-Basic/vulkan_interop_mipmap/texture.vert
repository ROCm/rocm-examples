#version 450

layout(location = 0) in vec2 inPosition;

layout(location = 0) out vec2 outUV;

void main()
{
    outUV = (inPosition + 1.0) / 2.0;
    gl_Position = vec4(inPosition, 0.0, 1.0);
}
