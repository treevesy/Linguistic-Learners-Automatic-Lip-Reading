from django.conf import settings
from django.urls import include, path, re_path
from django.contrib import admin

from wagtail.admin import urls as wagtailadmin_urls
from wagtail.core import urls as wagtail_urls
from wagtail.documents import urls as wagtaildocs_urls

from search import views as search_views
from cam_app import views, camera
from cam_app2 import views as v2
from lipreading import prediction
from django.http import StreamingHttpResponse, HttpResponse
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve 
from pathlib import Path
import json

deviceCamera = camera.VideoCamera();

urlpatterns = [

    path('django-admin/', admin.site.urls),

    path('admin/', include(wagtailadmin_urls)),
    path('documents/', include(wagtaildocs_urls)),

    path('search/', search_views.search, name='search'),

    path('scanner_video/', views.ScannerVideoView.as_view(), name='scanner_video'),
    path('img/', v2.ImageView.as_view(), name='img'),
    path('no_video/', views.NoVideoView.as_view(), name='no_video'),
    path('camera_feed/', lambda r: StreamingHttpResponse(camera.generate_frames(deviceCamera),
                                                     content_type='multipart/x-mixed-replace; boundary=frame;')),
    path('get_model_info', lambda r: HttpResponse(json.dumps(Path('lipreading/models/model_config.json').read_text()))),
    path('take_snippet/', lambda r: deviceCamera.createVideoSnippet(r), name='take_snippet'),
    #path('process_video/', lambda r: , name='process_video')
    re_path(r'^get_word_list/(?P<wordsListFileName>[0-9A-Za-z\.\/_-]+)', lambda r, wordsListFileName: HttpResponse(Path('lipreading/labels/' + wordsListFileName).read_text())),
    re_path(r'^predict_video/(?P<puType>[0-9])/(?P<numClasses>[0-9]+)/(?P<modelPath>[0-9A-Za-z\._-]+)/(?P<configPath>[0-9A-Za-z\._-]+)/(?P<wordListPath>[0-9A-Za-z\._-]+)/(?P<videoPath>[0-9A-Za-z\.\/_-]+)/$', lambda r, puType, numClasses, modelPath, configPath, wordListPath, videoPath: prediction.getPrediction(r, puType, numClasses, modelPath, configPath, wordListPath, videoPath), name='predict_video'),
    path('', include(wagtail_urls))

]


if settings.DEBUG:
    from django.conf.urls.static import static
    from django.contrib.staticfiles.urls import staticfiles_urlpatterns

    # Serve static and media files from development server
    urlpatterns += staticfiles_urlpatterns()
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
