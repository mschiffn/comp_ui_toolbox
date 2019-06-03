%
% superclass for all vibrating faces
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-06-18
%
classdef face

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties ( SetAccess = private )

        % independent properties
        apodization
        lens

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = face( varargin )

            if nargin == 0
                return;
            end

            objects = repmat( objects, size( varargin{ 1 } ) );
        end % function objects = face( varargin )

    end % methods

end % classdef face
