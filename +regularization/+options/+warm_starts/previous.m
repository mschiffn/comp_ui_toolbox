%
% superclass for all previous warm start options
%
% author: Martin F. Schiffner
% date: 2019-09-24
% modified: 2020-02-12
%
classdef previous < regularization.options.warm_starts.warm_start

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = previous( varargin )

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
            objects@regularization.options.warm_starts.warm_start( size );

        end % function objects = previous( size )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( warm_starts_previous )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.warm_starts.previous
            if ~isa( warm_starts_previous, 'regularization.options.warm_starts.previous' )
                errorStruct.message = 'warm_starts_previous must be regularization.options.warm_starts.previous!';
                errorStruct.identifier = 'string:NoOptionsWarmStartPrevious';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "previous"
            strs_out = repmat( "previous", size( warm_starts_previous ) );

        end % function strs_out = string( warm_starts_previous )

	end % methods

end % classdef previous < regularization.options.warm_starts.warm_start
