#include "cocoa-util.h"

#include <TargetConditionals.h>
#import <QuartzCore/CAMetalLayer.h>

#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#else
#import <Cocoa/Cocoa.h>
#endif

namespace rhi {

void* CocoaUtil::createMetalLayer(void* windowOrView)
{
    CAMetalLayer* layer = [CAMetalLayer layer];
#if TARGET_OS_IPHONE
    UIView* view = (UIView*)windowOrView;
    layer.frame = view.bounds;
    [view.layer addSublayer:layer];
#else
    NSWindow* window = (NSWindow*)windowOrView;
    window.contentView.layer = layer;
    window.contentView.wantsLayer = YES;
#endif
    return layer;
}

void* CocoaUtil::nextDrawable(void* metalLayer)
{
    CAMetalLayer* layer = (CAMetalLayer*)metalLayer;
    return [layer nextDrawable];
}

void CocoaUtil::destroyMetalLayer(void* metalLayer)
{
    CAMetalLayer* layer = (CAMetalLayer*)metalLayer;
    [layer release];
}

} // namespace rhi
