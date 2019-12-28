%
% superclass for all previous warm start options
%
% author: Martin F. Schiffner
% date: 2019-09-24
% modified: 2019-09-24
%
classdef warm_start_previous < optimization.options.warm_start

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = warm_start_previous( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers

            %--------------------------------------------------------------
            % 2.) create previous warm start options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@optimization.options.warm_start( size );

        end % function objects = warm_start_previous( size )

	end % methods

end % classdef warm_start_previous < optimization.options.warm_start
