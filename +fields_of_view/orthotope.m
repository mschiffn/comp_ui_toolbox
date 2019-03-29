%
% superclass for all fields of view with orthotope shape
%
% author: Martin F. Schiffner
% date: 2018-01-23
% modified: 2019-03-28
%
classdef orthotope < fields_of_view.field_of_view & math.orthotope

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = orthotope( varargin )

            %--------------------------------------------------------------
            % 1.) constructors of superclasses
            %--------------------------------------------------------------
            objects@math.orthotope( varargin{ : } );
            objects@fields_of_view.field_of_view( nargin );

        end % function objects = orthotope( varargin )

    end

end % classdef orthotope
