%
% superclass for all inactive warm start options
%
% author: Martin F. Schiffner
% date: 2019-09-24
% modified: 2020-01-03
%
classdef warm_start_off < regularization.options.warm_start

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = warm_start_off( varargin )

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
            % 2.) create inactive warm start options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.warm_start( size );

        end % function objects = warm_start_off( size )

	end % methods

end % classdef warm_start_off < regularization.options.warm_start
