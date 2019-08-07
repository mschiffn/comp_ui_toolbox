%
% superclass for all direct algorithm options
%
% author: Martin F. Schiffner
% date: 2019-07-30
% modified: 2019-08-03
%
classdef algorithm_direct < scattering.options.algorithm

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = algorithm_direct( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vectors for size
            % superclass ensures positive integers for size

            %--------------------------------------------------------------
            % 2.) create direct algorithm options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.algorithm( size );

        end % function objects = algorithm_direct( varargin )

	end % methods

end % classdef algorithm_direct < scattering.options.algorithm
