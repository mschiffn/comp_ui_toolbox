%
% superclass for all inactive warm start options
%
% author: Martin F. Schiffner
% date: 2019-09-24
% modified: 2020-02-12
%
classdef off < regularization.options.warm_starts.warm_start

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = off( varargin )

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
            objects@regularization.options.warm_starts.warm_start( size );

        end % function objects = off( size )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( options_off )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.warm_starts.off
            if ~isa( options_off, 'regularization.options.warm_starts.off' )
                errorStruct.message = 'options_off must be regularization.options.warm_starts.off!';
                errorStruct.identifier = 'string:NoOptionsWarmStartOff';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "off"
            strs_out = repmat( "off", size( options_off ) );

        end % function strs_out = string( options_off )

	end % methods

end % classdef off < regularization.options.warm_starts.warm_start
